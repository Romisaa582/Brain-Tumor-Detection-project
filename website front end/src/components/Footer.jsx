import React from 'react'
import { assets } from '../assets/assets'
import LOGO from '../assets/LOGO1.png';

const Footer = () => {
  return (
    <div className='md:mx-10'>
        <div className='flex flex-col sm:grid grid-cols-[3fr_1fr_1fr] gap-14 my-10 mt-40 text-sm'>
            <div>
                <img className='mb-5 w-40' src={LOGO} alt="" />
                <p className='w-full md:w-2/3 text-gray-600 leading-6'>Empowering healthcare with AI diagnostics.</p>
            </div>
            {/* -------------------Center Section */}
            <div>
                <p className='text-x1 font-medium mb-5'>Company</p>
                <ul className='flex flex-col gap-2 text-gray-600'>
                    <li>Home</li>
                    <li>About</li>
                    <li>Contact Us</li>
                    <li>Privacy policy</li>
                </ul>
            </div>
            {/* --------------------Right Section */}
            <div>
                <p className='text-x1 font-medium mb-5'>Get IN TOUCH</p>
                <ul className='flex flex-col gap-2 text-gray-600'>
                    <li>+20-10-234-455</li>
                    <li>Weal24@gmail.com</li>
                </ul>
            </div>
        </div>
        {/*-----------------Copy Right Text-------------- */}
        <div>
            <hr />
            <p className='py-5 text-sm text-center '>Copyright © 2024-2025 Tumor Tracker Company S.L. All rights reserved.</p>
        </div>
    </div>
  )
}

export default Footer